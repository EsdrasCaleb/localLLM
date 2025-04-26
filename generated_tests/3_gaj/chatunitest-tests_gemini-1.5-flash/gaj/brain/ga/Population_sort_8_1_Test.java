package brain.ga;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class Population_sort_8_1_Test {

    @Test
    void testSortEmpty() {
        Population population = new Population();
        population.sort();
        assertEquals(0, population.getSize());
    }

    // Helper class for testing
    static class Genome implements Comparable<Genome> {

        private int score;

        public Genome(int score) {
            this.score = score;
        }

        public int getScore() {
            return score;
        }

        @Override
        public int compareTo(Genome other) {
            return Integer.compare(this.score, other.score);
        }

        @Override
        public String toString() {
            return "Genome{" + "score=" + score + '}';
        }
    }

    static class Selector {
    }

    static class Evaluator {
    }
}
