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
