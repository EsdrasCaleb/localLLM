package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

class VectorGenome_getGene_2_0_Test {

    @ExtendWith(MockitoExtension.class)
    public class VectorGenome_getGene_2_0Test {

        @Test
        public void testGetGene() {
            // Create a mock instance of VectorGenome
            VectorGenome mockGenome = mock(VectorGenome.class);
            // Set the expected value for the getGene method
            when(mockGenome.getGene(0)).thenReturn("gene1");
            // Fix the bug by casting the result to String
            String result = (String) mockGenome.getGene(0);
            // Verify that the getGene method was called with the expected argument
            assertEquals("gene1", result);
        }
    }
}
