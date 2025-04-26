package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Genome_initialize_1_1_Test {

    private Genome genome;

    private Evaluator evaluator;

    @BeforeEach
    void setUp() {
        genome = new Genome();
        evaluator = Mockito.mock(Evaluator.class);
    }

    @Test
    void testInitialize() {
        genome.initialize();
        assertEquals(0, evaluator.evaluate(genome));
    }
}
