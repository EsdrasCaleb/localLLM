package brain.ga;

import java.util.Random;
import java.util.Vector;
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

    private GAEnumAllelesSet gaEnumAllelesSet;

    private Vector mockAlleles;

    @BeforeEach
    public void setUp() {
        gaEnumAllelesSet = new GAEnumAllelesSet();
        mockAlleles = Mockito.mock(Vector.class);
        gaEnumAllelesSet.setAlleles(mockAlleles);
    }

    @Test
    public void testSizeEmpty() {
        Mockito.when(mockAlleles.size()).thenReturn(0);
        int size = gaEnumAllelesSet.size();
        assertEquals(0, size);
    }

    @Test
    public void testSizeNonEmpty() {
        int expectedSize = 5;
        Mockito.when(mockAlleles.size()).thenReturn(expectedSize);
        int size = gaEnumAllelesSet.size();
        assertEquals(expectedSize, size);
    }
}
