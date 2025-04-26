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

public class GAEnumAllelesSet_allele_0_0_Test {

    @Test
    public void testAllele() {
        // Arrange
        GAEnumAllelesSet allelesSet = new GAEnumAllelesSet();
        Vector<Object> alleles = new Vector<>();
        alleles.add("Allele1");
        alleles.add("Allele2");
        alleles.add("Allele3");
        allelesSet.setAlleles(alleles);
        // Act
        Object result = allelesSet.allele();
        // Assert
        assertEquals("Allele1", result);
    }
}
