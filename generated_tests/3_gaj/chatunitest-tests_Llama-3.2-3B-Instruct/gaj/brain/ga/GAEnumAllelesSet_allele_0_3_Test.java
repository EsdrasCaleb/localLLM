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
public class GAEnumAllelesSet_allele_0_3_Test {

    @Test
    public void testAllele_ReturnsRandomAllele() {
        // Arrange
        GAEnumAllelesSet gaEnumAllelesSet = new GAEnumAllelesSet();
        Vector alleles = mock(Vector.class);
        when(alleles.size()).thenReturn(10);
        when(alleles.get(anyInt())).thenReturn("A");
        gaEnumAllelesSet.setAlleles(alleles);
        // Act
        Object allele = gaEnumAllelesSet.allele();
        // Assert
        assertNotNull(allele);
        assertEquals("A", allele);
    }

    @Test
    public void testAllele_ReturnsRandomAllele_WhenVectorHasMoreThanOneElement() {
        // Arrange
        GAEnumAllelesSet gaEnumAllelesSet = new GAEnumAllelesSet();
        Vector alleles = new Vector();
        for (int i = 0; i < 10; i++) {
            alleles.add("A");
        }
        when(alleles.size()).thenReturn(10);
        gaEnumAllelesSet.setAlleles(alleles);
        // Act
        Object allele = gaEnumAllelesSet.allele();
        // Assert
        assertNotNull(allele);
        assertTrue(alleles.contains(allele));
    }
}
