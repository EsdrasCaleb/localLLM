package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Features_toString_4_0_Test {

    @Test
    void testToString() {
        Features features = new Features();
        features.setFeature(new String[] { "Feature 1", "Feature 2" });
        String actual = features.toString();
        String expected = "# of Feature = 2\nFeature - Feature 1\nFeature - Feature 2";
        assertEquals(expected, actual);
    }
}
