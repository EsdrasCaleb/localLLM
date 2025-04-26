package brain.ga;

import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

class Population_sort_8_0_Test {

    private Population population;

    @BeforeEach
    void setUp() {
        population = new Population();
    }

    @Test
    void testSortWithEmptyGenoms() {
        // Arrange
        // No genomes added
        // Act
        population.sort();
        // Assert
        assertEquals(0, population.getSize());
    }

    @Test
    void testSortWithOneGenome() {
        // Arrange
        Genome genome = mock(Genome.class);
        // Fixed: Changed to Double
        when(genome.getScore()).thenReturn(10.0);
        addGenomeToPopulation(genome);
        // Act
        population.sort();
        // Assert
        assertEquals(1, population.getSize());
        assertEquals(genome, population.genoms.get(0));
    }

    @Test
    void testSortWithMultipleGenomes() {
        // Arrange
        Genome genome1 = mock(Genome.class);
        Genome genome2 = mock(Genome.class);
        Genome genome3 = mock(Genome.class);
        // Fixed: Changed to Double
        when(genome1.getScore()).thenReturn(30.0);
        // Fixed: Changed to Double
        when(genome2.getScore()).thenReturn(10.0);
        // Fixed: Changed to Double
        when(genome3.getScore()).thenReturn(20.0);
        addGenomeToPopulation(genome1);
        addGenomeToPopulation(genome2);
        addGenomeToPopulation(genome3);
        // Act
        population.sort();
        // Assert
        assertEquals(3, population.getSize());
        // should be sorted
        assertEquals(genome2, population.genoms.get(0));
        assertEquals(genome3, population.genoms.get(1));
        assertEquals(genome1, population.genoms.get(2));
    }

    private void addGenomeToPopulation(Genome genome) {
        try {
            Field field = Population.class.getDeclaredField("genoms");
            field.setAccessible(true);
            List<Genome> genomes = (List<Genome>) field.get(population);
            genomes.add(genome);
        } catch (Exception e) {
            fail("Failed to add genome to population: " + e.getMessage());
        }
    }
}
