package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BlendedSearch_toString_3_0_Test {

    @Test
    void testToString() {
        BlendedSearch blendedSearch = new BlendedSearch();
        blendedSearch.setProductLine(new ProductLine[] {});
        String expectedOutput = "productLines is null \n";
        String actualOutput = blendedSearch.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
