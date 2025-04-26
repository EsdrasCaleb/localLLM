package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BlendedSearch_printProductList_4_0_Test {

    @Mock
    private BlendedSearch blendedSearch;

    @InjectMocks
    private BlendedSearch blendedSearchUnderTest;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testPrintProductListNull() {
        when(blendedSearch.getProductLinesArrayList()).thenReturn(null);
        String result = blendedSearchUnderTest.printProductList();
        assertEquals("productLines is null \n", result);
    }
}
