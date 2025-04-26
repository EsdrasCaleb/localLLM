package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class Population_selectNextGenome_1_0_Test {

    @Mock
    private Selector selector;

    @Mock
    private Evaluator evaluator;

    @InjectMocks
    private Population population;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testSelectNextGenome() {
        Genome mockGenome = mock(Genome.class);
        when(selector.select(population)).thenReturn(mockGenome);
        Genome result = population.selectNextGenome();
        assertNotNull(result);
        assertEquals(mockGenome, result);
        verify(selector, times(1)).select(population);
    }
}
