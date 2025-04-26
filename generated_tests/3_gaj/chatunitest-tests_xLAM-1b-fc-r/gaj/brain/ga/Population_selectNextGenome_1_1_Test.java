package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
public class Population_selectNextGenome_1_1_Test {

    @Mock
    private Selector selector;

    @Mock
    private Evaluator evaluator;

    @InjectMocks
    private Population population;

    @Test
    public void testSelectNextGenome() {
        // Given
        when(selector.select(population)).thenReturn(new Genome());
        // When
        Genome genome = population.selectNextGenome();
        // Then
        assertEquals(new Genome(), genome);
    }
}
