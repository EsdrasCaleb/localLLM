package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class Population_get_6_0_Test {

    private Population population;

    private Genome genome1;

    private Genome genome2;

    @BeforeEach
    void setUp() {
        population = new Population();
        // Dummy genome
        genome1 = new Genome();
        genome2 = new Genome();
        List<Genome> genomes = new ArrayList<>();
        genomes.add(genome1);
        genomes.add(genome2);
        try {
            java.lang.reflect.Field genomsField = Population.class.getDeclaredField("genoms");
            genomsField.setAccessible(true);
            genomsField.set(population, genomes);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
            fail("Error accessing private field");
        }
    }

    @Test
    void testGetValidIndex() {
        Genome retrievedGenome = population.get(2);
        assertEquals(genome2, retrievedGenome);
    }

    @Test
    void testGetInvalidIndex() {
        assertThrows(IndexOutOfBoundsException.class, () -> population.get(0));
    }

    @Test
    void testGetIndexZero() {
        assertThrows(IndexOutOfBoundsException.class, () -> population.get(1));
    }

    @Test
    void testGetIndexGreaterThanSize() {
        assertThrows(IndexOutOfBoundsException.class, () -> population.get(3));
    }

    // Example of a test with an empty list
    @Test
    void testGetEmptyList() {
        Population emptyPopulation = new Population();
        assertThrows(IndexOutOfBoundsException.class, () -> emptyPopulation.get(1));
    }
}

// Dummy class for Genome
class Genome {

    // Add necessary fields and methods to Genome if needed
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        return true;
    }
}
