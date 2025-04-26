package net.kencochrane.a4j.beans;

import java.lang.reflect.InvocationTargetException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BlendedSearch_printProductList_4_2_Test {

    @Test
    public void testPrintProductList() throws NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException {
        BlendedSearch blendedSearch = new BlendedSearch();
        blendedSearch.setProductLine(new ProductLine[0]);
        String expectedOutput = "# of productLines = 0\n";
        String actualOutput = blendedSearch.printProductList();
        assertEquals(expectedOutput, actualOutput);
    }
}
