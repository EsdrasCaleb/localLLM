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

class GAEnumAllelesSet_allele_1_2_Test {

    @Test
    public void testAllele() throws Exception {
        // Arrange
        GAEnumAllelesSet gaEnumAllelesSet = new GAEnumAllelesSet();
        Vector alleles = new Vector<>();
        alleles.add("A");
        alleles.add("B");
        alleles.add("C");
        gaEnumAllelesSet.setAlleles(alleles);
        // Act
        Object result = gaEnumAllelesSet.allele(1);
        // Assert
        assertEquals("B", result);
    }
}
