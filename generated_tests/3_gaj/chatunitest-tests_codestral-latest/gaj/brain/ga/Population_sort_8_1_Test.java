package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

class Population_sort_8_1_Test {

    @InjectMocks
    private Population population;

    @Mock
    private Selector selector;

    @Mock
    private Evaluator evaluator;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        population.genoms = new ArrayList<>();
    }

    @Test
    void testSort() {
        Genome genome1 = mock(Genome.class);
        when(genome1.getScore()).thenReturn(10.0);
        Genome genome2 = mock(Genome.class);
        when(genome2.getScore()).thenReturn(5.0);
        Genome genome3 = mock(Genome.class);
        when(genome3.getScore()).thenReturn(15.0);
        population.genoms.add(genome1);
        population.genoms.add(genome2);
        population.genoms.add(genome3);
        // Act
        population.sort();
        // Assert
        List<Genome> sortedGenoms = new ArrayList<>(population.genoms);
        Collections.sort(sortedGenoms, Comparator.comparingDouble(Genome::getScore).reversed());
        assertEquals(sortedGenoms, population.genoms);
    }
}
