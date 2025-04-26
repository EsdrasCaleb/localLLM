package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

public class VectorGenome_getGene_2_2_Test {

    @Test
    void testGetGene() {
        // Arrange
        VectorGenome vectorGenome = Mockito.mock(VectorGenome.class);
        vectorGenome.setGene(0, "Gene1");
        vectorGenome.setGene(1, "Gene2");
        vectorGenome.setGene(2, "Gene3");
        // Repair the buggy line: compatible type
        String gene = (String) vectorGenome.getGene(1);
        // Assert
        assertEquals("Gene2", gene);
    }
}
