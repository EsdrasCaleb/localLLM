package brain.ga;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import java.util.Random;
import static org.junit.Assert.assertEquals;
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

@RunWith(MockitoJUnitRunner.class)
public class GAEnumAllelesSet_allele_0_0_Test {

    @Mock
    private Random random;

    @InjectMocks
    private GAEnumAllelesSet gaEnumAllelesSet;

    @Test
    public void testAllele() {
        // Arrange
        Vector alleles = new Vector();
        alleles.add("A");
        alleles.add("B");
        alleles.add("C");
        when(random.nextInt(alleles.size())).thenReturn(0);
        // Act
        Object allele = gaEnumAllelesSet.allele();
        // Assert
        assertEquals("A", allele);
    }
}
