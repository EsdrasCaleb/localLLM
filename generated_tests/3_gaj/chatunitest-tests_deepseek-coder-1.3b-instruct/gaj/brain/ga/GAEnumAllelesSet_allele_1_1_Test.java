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

public class GAEnumAllelesSet_allele_1_1_Test {

    @Test
    public void testAllele() {
        // Arrange
        Vector mockAlleles = new Vector();
        mockAlleles.add("Allele1");
        mockAlleles.add("Allele2");
        mockAlleles.add("Allele3");
        GAEnumAllelesSet allelesSet = new GAEnumAllelesSet();
        allelesSet.setAlleles(mockAlleles);
        // Act
        Object result = allelesSet.allele(1);
        // Assert
        assertEquals("Allele2", result);
    }
}
