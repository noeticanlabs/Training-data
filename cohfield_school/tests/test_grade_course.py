import unittest
from cohfield_school.grade_course import GradeCourse
from cohfield_school.grade_worlds import WORLDS


class GradeCourseTests(unittest.TestCase):
    def test_each_course_aggregates_three_worlds_without_certification(self):
        for grade in WORLDS:
            with self.subTest(grade=grade):
                course = GradeCourse(grade, "cohort-001")
                self.assertFalse(course.report()["all_worlds_examined"])
                self.assertIsNone(course.report()["provisional_accuracy"])
                for world in course.worlds:
                    train = course.training(world)
                    for probe in train["allowed_probes"]:
                        course.intervene(world, probe)
                    course.exam(world)
                    course.submit(world, None)
                report = course.report()
                self.assertTrue(report["all_worlds_examined"])
                self.assertEqual(report["examined_worlds"], 3)
                self.assertIsNotNone(report["provisional_accuracy"])
                self.assertFalse(report["certified"])
                self.assertEqual(report["grade_status"], "UNPROMOTED_SCHOOL_EXERCISE")

    def test_no_wrong_world_or_double_submission(self):
        course = GradeCourse("G0", "seed")
        with self.assertRaises(ValueError):
            course.training("wrong")
        world = course.worlds[0]
        course.exam(world)
        with self.assertRaises(ValueError):
            course.exam(world)
        course.submit(world, "state_changed")
        with self.assertRaises(ValueError):
            course.submit(world, "state_changed")


if __name__ == "__main__":
    unittest.main()
