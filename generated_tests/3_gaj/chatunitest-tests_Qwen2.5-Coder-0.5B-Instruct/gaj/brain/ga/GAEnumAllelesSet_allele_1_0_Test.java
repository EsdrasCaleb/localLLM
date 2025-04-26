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

class GAEnumAllelesSet_allele_1_0_Test {

    @Test
    public void testAllele() {
        // Create an instance of GAEnumAllelesSet
        GAEnumAllelesSet gaEnumAllelesSet = new GAEnumAllelesSet();
        // Create a vector of alleles
        Vector<Integer> alleles = new Vector<>();
        alleles.add(1);
        alleles.add(2);
        alleles.add(3);
        // Set the alleles in the GAEnumAllelesSet instance
        gaEnumAllelesSet.setAlleles(alleles);
        // Call the allele method with an index
        Object result = gaEnumAllelesSet.allele(2);
        // Verify that the result is the expected value
        assertEquals(2, result);
    }
}
