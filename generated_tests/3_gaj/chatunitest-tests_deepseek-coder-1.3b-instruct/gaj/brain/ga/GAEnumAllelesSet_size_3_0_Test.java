package brain.ga;

import java.util.Vector;
import java.util.Random;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // and returns one of them
java.util.*;

// The allele set class is a container for the different values that a gene may assume.
// If you call the allele member function with no argument,
// the allele set picks randomly from the alleles it contains
public class GAEnumAllelesSet_size_3_0_Test {

    private GAEnumAllelesSet focalObject;

    private Vector mockAlleles;

    private Random mockRandom;

    @BeforeEach
    public void setUp() {
        mockAlleles = Mockito.mock(Vector.class);
        mockRandom = Mockito.mock(Random.class);
        focalObject = new GAEnumAllelesSet();
        focalObject.setAlleles(mockAlleles);
    }

    @Test
    public void testSize() {
        // Given
        int expectedSize = 5;
        Mockito.when(mockAlleles.size()).thenReturn(expectedSize);
        // When
        int actualSize = focalObject.size();
        // Then
        assertEquals(expectedSize, actualSize);
    }
}
