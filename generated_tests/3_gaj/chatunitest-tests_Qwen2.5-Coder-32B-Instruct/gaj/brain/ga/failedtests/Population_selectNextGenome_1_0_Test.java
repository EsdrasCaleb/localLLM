package brain.ga;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class Population_selectNextGenome_1_0_Test {

    @Mock
    private Selector selector;

    @Mock
    private Evaluator evaluator;

    @Mock
    private Genome genome;

    @InjectMocks
    private Population population;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        List<Genome> genoms = new ArrayList<>();
        genoms.add(genome);
        Field field = Population.class.getDeclaredField("genoms");
        field.setAccessible(true);
        field.set(population, genoms);
    }

    @Test
    public void testSelectNextGenome_WithSelector() {
        when(selector.select(population)).thenReturn(genome);
        Genome selectedGenome = population.selectNextGenome();
        assertNotNull(selectedGenome);
        assertEquals(genome, selectedGenome);
        verify(selector, times(1)).select(population);
    }

    @Test
    public void testSelectNextGenome_WithoutSelector() {
        population.setSelector(null);
        Genome selectedGenome = population.selectNextGenome();
        assertNull(selectedGenome);
        verify(selector, never()).select(population);
    }
}
