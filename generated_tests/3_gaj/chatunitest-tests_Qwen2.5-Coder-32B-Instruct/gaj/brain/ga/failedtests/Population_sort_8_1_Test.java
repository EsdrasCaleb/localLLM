package brain.ga;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class Population_sort_8_1_Test {

    @Mock
    private Selector selector;

    @Mock
    private Evaluator evaluator;

    private Population population;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        population = new Population();
        population.setSelector(selector);
        population.setEvaluator(evaluator);
    }

    @Test
    void testSortWithEmptyGenoms() throws Exception {
        // Arrange
        List<Genome> genoms = new ArrayList<>();
        setPrivateField(population, "genoms", genoms);
        // Act
        population.sort();
        // Assert
        assertTrue(genoms.isEmpty());
        verifyNoInteractions(selector, evaluator);
    }

    @Test
    void testSortWithNonEmptyGenoms() throws Exception {
        // Arrange
        List<Genome> genoms = new ArrayList<>();
        Genome genome1 = mock(Genome.class);
        Genome genome2 = mock(Genome.class);
        when(genome1.compareTo(genome2)).thenReturn(-1);
        when(genome1.getScore()).thenReturn(10.0);
        when(genome2.getScore()).thenReturn(20.0);
        genoms.add(genome2);
        genoms.add(genome1);
        setPrivateField(population, "genoms", genoms);
        // Act
        population.sort();
        // Assert
        assertEquals(genome1, genoms.get(0));
        assertEquals(genome2, genoms.get(1));
        verifyNoInteractions(selector, evaluator);
    }

    private void setPrivateField(Object target, String fieldName, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }
}

class Genome implements Comparable<Genome> {

    private double score;

    public Genome(double score) {
        this.score = score;
    }

    public double getScore() {
        return score;
    }

    @Override
    public int compareTo(Genome other) {
        return Double.compare(this.score, other.score);
    }

    @Override
    public String toString() {
        return "Genome{" + "score=" + score + '}';
    }
}

interface Selector {
}

interface Evaluator {
}
