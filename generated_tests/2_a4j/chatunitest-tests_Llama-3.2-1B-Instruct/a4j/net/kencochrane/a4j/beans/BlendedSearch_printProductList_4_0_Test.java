package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BlendedSearch_printProductList_4_0_Test {

    @Test
    public void testPrintProductList() {
        BlendedSearch blendedSearch = new BlendedSearch();
        String expectedOutput = "ProductList: \n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n";
        String actualOutput = blendedSearch.printProductList();
        assertEquals(expectedOutput, actualOutput);
    }
}
