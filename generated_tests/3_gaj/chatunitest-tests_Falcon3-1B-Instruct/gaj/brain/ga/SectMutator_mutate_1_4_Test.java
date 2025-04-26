package brain.ga;

import org.junit.Test;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class SectMutator_mutate_1_4_Test {

    @Test
    public void testMutateMethod() {
        // Create a new instance of SectMutator
        SectMutator sectMutator = new SectMutator();
        // Set the allelesSet to a known value
        sectMutator.setAllelesSet(new GAEnumAllelesSet());
        // Call the mutate method with a mutation probability of 0.5
        int mutCount = sectMutator.mutate(new Genome(), 0.5);
        // Assert the number of mutations performed
        assertEquals(1, mutCount);
        // Create another instance of SectMutator
        sectMutator.setAllelesSet(new GAEnumAllelesSet());
        // Call the mutate method again with the same mutation probability
        int mutCount2 = sectMutator.mutate(new Genome(), 0.5);
        // Assert the number of mutations performed remains the same
        assertEquals(1, mutCount2);
        // Create a new instance of SectMutator
        sectMutator.setAllelesSet(new GAEnumAllelesSet());
        // Call the mutate method with a different mutation probability
        int mutCount3 = sectMutator.mutate(new Genome(), 0.8);
        // Assert the number of mutations performed does not change
        assertEquals(0, mutCount3);
        // Create a new instance of SectMutator
        sectMutator.setAllelesSet(new GAEnumAllelesSet());
        // Call the mutate method with a different mutation probability
        int mutCount4 = sectMutator.mutate(new Genome(), 0.2);
        // Assert the number of mutations performed does not change
        assertEquals(0, mutCount4);
    }
}
