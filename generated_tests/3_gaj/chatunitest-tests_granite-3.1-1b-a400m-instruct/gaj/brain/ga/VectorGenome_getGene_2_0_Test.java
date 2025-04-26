package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class VectorGenome_getGene_2_0_Test {

    @Test
    void testGetGeneOutOfBounds() {
        VectorGenome vectorGenome = new VectorGenome();
        assertThrows(IndexOutOfBoundsException.class, () -> vectorGenome.getGene(Integer.MAX_VALUE));
    }
}
