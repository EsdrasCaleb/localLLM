package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class RankSelector_select_0_0_Test {

    private RankSelector rankSelector;

    private Population population;

    @BeforeEach
    void setUp() {
        rankSelector = new RankSelector();
        population = mock(Population.class);
    }

    @Test
    void testSelectReturnsGenome() {
        // Arrange
        Genome genome1 = new Genome();
        Genome genome2 = new Genome();
        List<Genome> genomes = Arrays.asList(genome1, genome2);
        when(population.getSize()).thenReturn(genomes.size());
        when(population.get(0)).thenReturn(genome1);
        when(population.get(1)).thenReturn(genome2);
        // Mocking GAUtilities.nextPos to return 0 and 1 for testing
        try {
            java.lang.reflect.Field field = GAUtilities.class.getDeclaredField("random");
            field.setAccessible(true);
            // Set seed for reproducibility
            field.set(null, new java.util.Random(0));
        } catch (Exception e) {
            fail("Failed to set random field: " + e.getMessage());
        }
        // Act
        Genome selectedGenome1 = rankSelector.select(population);
        Genome selectedGenome2 = rankSelector.select(population);
        // Assert
        assertEquals(genome1, selectedGenome1);
        assertEquals(genome2, selectedGenome2);
    }

    @Test
    void testSelectEmptyPopulation() {
        // Arrange
        when(population.getSize()).thenReturn(0);
        // Act & Assert
        assertThrows(IndexOutOfBoundsException.class, () -> rankSelector.select(population));
    }
}
