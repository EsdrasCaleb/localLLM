package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class Genome_initialize_1_3_Test {

    // Test class
    @Test
    public void testInitialize() {
        Genome genome = new Genome();
        genome.initialize();
        // assert that the genome was initialized
    }
}
