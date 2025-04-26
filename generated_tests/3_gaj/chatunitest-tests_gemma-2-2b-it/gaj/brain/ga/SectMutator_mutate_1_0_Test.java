package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class SectMutator_mutate_1_0_Test {

    @Test
    void mutate() {
        SectMutator mutator = new SectMutator();
        Genome genome = new VectorGenome();
        double pmut = 0.2;
        int result = mutator.mutate(genome, pmut);
        assertEquals(0, result);
    }
}
