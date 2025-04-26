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
        genome1 = mock(Genome.class);
        genome2 = mock(Genome.class);
        // Adding mock genomes to the population
        population.genoms.add(genome1);
        population.genoms.add(genome2);
    }

    @Test
    void testGet_ValidIndex() {
        // Test for valid index 1
        assertEquals(genome1, population.get(1));
        // Test for valid index 2
        assertEquals(genome2, population.get(2));
    }

    @Test
    void testGet_IndexOutOfBounds() {
        // Test for index less than 1
        assertThrows(IndexOutOfBoundsException.class, () -> population.get(0));
        // Test for index greater than size
        assertThrows(IndexOutOfBoundsException.class, () -> population.get(3));
    }
}
