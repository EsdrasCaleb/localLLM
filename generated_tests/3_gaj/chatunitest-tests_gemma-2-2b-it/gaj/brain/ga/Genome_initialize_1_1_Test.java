package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Genome_initialize_1_1_Test {

    @Test
    void initialize() {
        Genome genome = new Genome();
        genome.initialize();
        assertTrue(genome.getEvaluator() instanceof Evaluator);
    }
}
