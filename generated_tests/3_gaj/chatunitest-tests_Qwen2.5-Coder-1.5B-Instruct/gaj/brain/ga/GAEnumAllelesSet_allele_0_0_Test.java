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

class GAEnumAllelesSet_allele_0_0_Test {

    @Test
    public void testAllele() {
        // Arrange
        GAEnumAllelesSet gaEnumAllelesSet = new GAEnumAllelesSet();
        Vector<String> alleles = new Vector<>();
        alleles.add("A");
        alleles.add("B");
        alleles.add("C");
        gaEnumAllelesSet.setAlleles(alleles);
        // Act
        Object result = gaEnumAllelesSet.allele();
        // Assert
        assertNotNull(result);
        assertTrue(alleles.contains(result.toString()));
    }
}
