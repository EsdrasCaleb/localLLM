package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BlendedSearch_toString_3_0_Test {

    @Test
    public void testToString() {
        BlendedSearch blendedSearch = new BlendedSearch();
        blendedSearch.setProductLine(new ProductLine[] { new ProductLine(), new ProductLine() });
        String expectedOutput = "productLines is null \n# of productLines = 2\n";
        assertEquals(expectedOutput, blendedSearch.toString());
    }
}
