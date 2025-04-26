package brain.ga;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class Population_get_6_2_Test {

    private Population population;

    private List<Genome> genomes;

    @BeforeEach
    void setUp() {
        population = new Population();
        genomes = new ArrayList<>();
        genomes.add(new Genome());
        genomes.add(new Genome());
        genomes.add(new Genome());
        try {
            Field genomsField = Population.class.getDeclaredField("genoms");
            genomsField.setAccessible(true);
            genomsField.set(population, genomes);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set genoms field: " + e.getMessage());
        }
    }

    @Test
    void testGetValidIndex() {
        Genome genome = population.get(2);
        assertNotNull(genome);
    }

    @Test
    void testGetIndexZero() {
        assertThrows(IndexOutOfBoundsException.class, () -> population.get(0));
    }

    @Test
    void testGetIndexTooLarge() {
        assertThrows(IndexOutOfBoundsException.class, () -> population.get(4));
    }

    @Test
    void testGetIndexNegative() {
        assertThrows(IndexOutOfBoundsException.class, () -> population.get(-1));
    }

    @Test
    void testGetEmptyPopulation() {
        Population emptyPopulation = new Population();
        assertThrows(IndexOutOfBoundsException.class, () -> emptyPopulation.get(1));
    }
}

class Population {

    private List<Genome> genoms;

    public Population() {
        this.genoms = new ArrayList<>();
    }

    public Genome get(int index) {
        if (index < 1 || index > genoms.size()) {
            throw new IndexOutOfBoundsException("Index out of bounds");
        }
        return genoms.get(index - 1);
    }
}

class Genome {
}
