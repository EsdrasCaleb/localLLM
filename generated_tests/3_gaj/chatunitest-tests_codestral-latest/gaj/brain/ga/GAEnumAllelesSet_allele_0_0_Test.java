package brain.ga;

import java.util.Random;
import java.util.Vector;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // and returns one of them
java.util.*;
// The allele set class is a container for the different values that a gene may assume.
// If you call the allele member function with no argument,
// the allele set picks randomly from the alleles it contains

@ExtendWith(MockitoExtension.class)
public class GAEnumAllelesSet_allele_0_0_Test {

    @Mock
    private Random rnd;

    @InjectMocks
    private GAEnumAllelesSet gaEnumAllelesSet;

    private Vector<Object> alleles;

    @BeforeEach
    public void setUp() {
        alleles = new Vector<>();
        alleles.add("allele1");
        alleles.add("allele2");
        alleles.add("allele3");
        gaEnumAllelesSet.setAlleles(alleles);
    }

    @Test
    public void testAllele() {
        when(rnd.nextInt(alleles.size())).thenReturn(0);
        assertEquals("allele1", gaEnumAllelesSet.allele());
        when(rnd.nextInt(alleles.size())).thenReturn(1);
        assertEquals("allele2", gaEnumAllelesSet.allele());
        when(rnd.nextInt(alleles.size())).thenReturn(2);
        assertEquals("allele3", gaEnumAllelesSet.allele());
    }
}
