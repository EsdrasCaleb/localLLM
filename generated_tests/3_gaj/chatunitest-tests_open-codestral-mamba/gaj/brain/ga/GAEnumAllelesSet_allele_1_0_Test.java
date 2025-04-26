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

public class GAEnumAllelesSet_allele_1_0_Test {

    @Mock
    private Vector alleles;

    @InjectMocks
    private GAEnumAllelesSet gaEnumAllelesSet;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testAllele() {
        Object expectedAllele = new Object();
        when(alleles.get(0)).thenReturn(expectedAllele);
        Object actualAllele = gaEnumAllelesSet.allele(0);
        assertEquals(expectedAllele, actualAllele);
    }
}
