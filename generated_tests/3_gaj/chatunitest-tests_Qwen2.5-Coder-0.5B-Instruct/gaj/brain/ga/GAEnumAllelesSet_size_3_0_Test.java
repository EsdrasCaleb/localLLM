package brain.ga;

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

class GAEnumAllelesSet_size_3_0_Test {

    @Test
    public void testSize() {
        // Arrange
        Random rnd = mock(Random.class);
        Vector alleles = mock(Vector.class);
        when(alleles.size()).thenReturn(10);
        // Act
        GAEnumAllelesSet instance = new GAEnumAllelesSet();
        instance.setAlleles(alleles);
        // Assert
        assertEquals(10, instance.size());
    }
}
