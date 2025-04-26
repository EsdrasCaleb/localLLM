package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class Population_get_6_0_Test {

    @InjectMocks
    private Population population;

    @Mock
    private Selector selector;

    @Mock
    private Evaluator evaluator;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        population.genoms = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            Genome genome = mock(Genome.class);
            population.genoms.add(genome);
        }
    }

    @Test
    public void testGet() {
        Genome genome = population.get(1);
        assertNotNull(genome);
        assertEquals(population.genoms.get(0), genome);
        genome = population.get(3);
        assertNotNull(genome);
        assertEquals(population.genoms.get(2), genome);
        assertThrows(IndexOutOfBoundsException.class, () -> population.get(6));
    }
}
