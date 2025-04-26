package brain.ga;

import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
class RankSelector_select_0_0_Test {

    @Mock
    private Population population;

    @Test
    void testSelect_PopulationNotEmpty() throws NoSuchFieldException, IllegalAccessException {
        // Create a mock Population
        List<Genome> genomes = new ArrayList<>();
        genomes.add(new Genome("genome1"));
        genomes.add(new Genome("genome2"));
        genomes.add(new Genome("genome3"));
        when(population.getSize()).thenReturn(genomes.size());
        when(population.get(anyInt())).thenAnswer(invocation -> genomes.get(invocation.getArgument(0)));
        // Create a RankSelector instance
        RankSelector selector = new RankSelector();
        // Mock GAUtilities.nextPos to return a specific index (to test different branches)
        when(GAUtilities.nextPos(anyInt())).thenReturn(0);
        Genome selectedGenome = selector.select(population);
        assertEquals("genome1", selectedGenome.getData());
        when(GAUtilities.nextPos(anyInt())).thenReturn(1);
        selectedGenome = selector.select(population);
        assertEquals("genome2", selectedGenome.getData());
        when(GAUtilities.nextPos(anyInt())).thenReturn(2);
        selectedGenome = selector.select(population);
        assertEquals("genome3", selectedGenome.getData());
        // Verify that get method was called with correct index
        verify(population, times(3)).get(anyInt());
    }

    @Test
    void testSelect_EmptyPopulation() {
        when(population.getSize()).thenReturn(0);
        RankSelector selector = new RankSelector();
        assertThrows(IndexOutOfBoundsException.class, () -> selector.select(population));
    }

    // Dummy Genome and Population classes for testing
    static class Genome {

        private String data;

        Genome(String data) {
            this.data = data;
        }

        public String getData() {
            return data;
        }
    }

    interface Population {

        int getSize();

        Genome get(int index);
    }

    static class GAUtilities {

        public static int nextPos(int size) {
            // Replace with actual implementation for testing purposes.
            return 0;
        }
    }

    static class RankSelector {

        public Genome select(Population population) {
            int size = population.getSize();
            if (size == 0)
                throw new IndexOutOfBoundsException();
            int index = GAUtilities.nextPos(size);
            return population.get(index);
        }
    }
}
