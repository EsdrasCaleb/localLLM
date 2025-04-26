package brain.ga;

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

class GAEnumAllelesSet_allele_1_0_Test {

    @InjectMocks
    private GAEnumAllelesSet gaEnumAllelesSet;

    @Mock
    private Vector alleles;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        gaEnumAllelesSet.setAlleles(alleles);
    }

    @Test
    void testAllele() {
        // Arrange
        int index = 0;
        Object expectedAllele = "allele";
        when(alleles.get(index)).thenReturn(expectedAllele);
        // Act
        Object result = gaEnumAllelesSet.allele(index);
        // Assert
        assertEquals(expectedAllele, result);
        verify(alleles).get(index);
    }

    @Test
    void testAllele_IndexOutOfBounds() {
        // Arrange
        int index = 1;
        when(alleles.get(index)).thenThrow(new ArrayIndexOutOfBoundsException());
        // Act & Assert
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> gaEnumAllelesSet.allele(index));
        verify(alleles).get(index);
    }
}
