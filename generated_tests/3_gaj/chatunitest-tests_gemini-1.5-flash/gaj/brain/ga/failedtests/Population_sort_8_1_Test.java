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

    @Test
    void testSortOneElement() {
        Population population = new Population();
        Genome genome = mock(Genome.class);
        when(genome.getScore()).thenReturn(10);
        population.genoms.add(genome);
        population.sort();
        assertEquals(1, population.getSize());
    }

    @Test
    void testSortMultipleElements() {
        Population population = new Population();
        Genome genome1 = mock(Genome.class);
        when(genome1.getScore()).thenReturn(20);
        Genome genome2 = mock(Genome.class);
        when(genome2.getScore()).thenReturn(10);
        Genome genome3 = mock(Genome.class);
        when(genome3.getScore()).thenReturn(15);
        population.genoms.add(genome1);
        population.genoms.add(genome2);
        population.genoms.add(genome3);
        population.sort();
        assertEquals(3, population.getSize());
        // Verify order (This part is tricky without access to Genome's internal comparison)
        // We can only check if the order is consistent with scores if Genome implements Comparable or Comparator is used.
        // Assuming Genome implements Comparable based on score.
        List<Genome> sortedGenomes = population.genoms;
        assertEquals(10, sortedGenomes.get(0).getScore());
        assertEquals(15, sortedGenomes.get(1).getScore());
        assertEquals(20, sortedGenomes.get(2).getScore());
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
