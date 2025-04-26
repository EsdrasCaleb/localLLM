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
public class GAEnumAllelesSet_allele_0_0_Test {

    private GAEnumAllelesSet gaEnumAllelesSet;

    @BeforeEach
    public void setUp() {
        gaEnumAllelesSet = new GAEnumAllelesSet();
    }

    @Test
    public void testAllelWithSingleAllele() {
        Vector<Object> alleles = new Vector<>();
        Object allele = new Object();
        alleles.add(allele);
        gaEnumAllelesSet.setAlleles(alleles);
        Object result = gaEnumAllelesSet.allele();
        // Should always return the single allele
        assertTrue(result == allele);
    }

    @Test
    public void testAllelWithMultipleAlleles() {
        Vector<Object> alleles = new Vector<>();
        Object allele1 = new Object();
        Object allele2 = new Object();
        alleles.add(allele1);
        alleles.add(allele2);
        gaEnumAllelesSet.setAlleles(alleles);
        // Call the allele() method multiple times to check if it returns either allele
        boolean foundAllele1 = false;
        boolean foundAllele2 = false;
        for (int i = 0; i < 100; i++) {
            Object result = gaEnumAllelesSet.allele();
            if (result == allele1) {
                foundAllele1 = true;
            } else if (result == allele2) {
                foundAllele2 = true;
            }
        }
        // We should have found both alleles in the results
        assertTrue(foundAllele1);
        assertTrue(foundAllele2);
    }
}
