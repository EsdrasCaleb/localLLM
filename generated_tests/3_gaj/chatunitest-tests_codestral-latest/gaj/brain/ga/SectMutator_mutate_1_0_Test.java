package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
class SectMutator_mutate_1_0_Test {

    @Mock
    private Random rnd;

    @InjectMocks
    private GAEnumAllelesSet allelesSet;

    @BeforeEach
    void setUp() {
        Vector<Object> alleles = new Vector<>();
        alleles.add("A");
        alleles.add("B");
        alleles.add("C");
        allelesSet.setAlleles(alleles);
    }

    @Test
    void testAllele() {
        when(rnd.nextInt(anyInt())).thenReturn(1);
        assertEquals("B", allelesSet.allele());
    }

    @Test
    void testAlleleWithIndex() {
        assertEquals("A", allelesSet.allele(0));
        assertEquals("B", allelesSet.allele(1));
        assertEquals("C", allelesSet.allele(2));
    }

    @Test
    void testSetAlleles() {
        Vector<Object> newAlleles = new Vector<>();
        newAlleles.add("X");
        newAlleles.add("Y");
        allelesSet.setAlleles(newAlleles);
        assertEquals(2, allelesSet.size());
        assertEquals("X", allelesSet.allele(0));
        assertEquals("Y", allelesSet.allele(1));
    }

    @Test
    void testSize() {
        assertEquals(3, allelesSet.size());
    }
}
