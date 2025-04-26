package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class BlendedSearch_toString_3_3_Test {

    // Test class
    @Test
    public void testToString() {
        BlendedSearch obj = new BlendedSearch();
        String expected = "productLines is null \n# of productLines = 0\n";
        String actual = obj.toString();
        assertEquals(expected, actual);
    }
}
