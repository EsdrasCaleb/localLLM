package brain.ga;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class Population_get_6_0_Test {

    private Population population;

    private List<Genome> mockGenoms;

    @BeforeEach
    void setUp() throws Exception {
        population = new Population();
        mockGenoms = new ArrayList<>();
        mockGenoms.add(new Genome());
        mockGenoms.add(new Genome());
        mockGenoms.add(new Genome());
        Field genomsField = Population.class.getDeclaredField("genoms");
        genomsField.setAccessible(true);
        genomsField.set(population, mockGenoms);
    }

    @Test
    void testGetWithValidIndex() {
        assertEquals(mockGenoms.get(0), population.get(1));
        assertEquals(mockGenoms.get(1), population.get(2));
        assertEquals(mockGenoms.get(2), population.get(3));
    }

    @Test
    void testGetWithInvalidIndex() {
        Exception exception = assertThrows(IndexOutOfBoundsException.class, () -> {
            population.get(0);
        });
        assertNotNull(exception);
        exception = assertThrows(IndexOutOfBoundsException.class, () -> {
            population.get(4);
        });
        assertNotNull(exception);
    }
}

class Genome {
    // Genome class can be left as is or can have additional fields and methods
}

interface Selector {
    // Selector interface can be left as is or can have additional methods
}

interface Evaluator {
    // Evaluator interface can be left as is or can have additional methods
}
