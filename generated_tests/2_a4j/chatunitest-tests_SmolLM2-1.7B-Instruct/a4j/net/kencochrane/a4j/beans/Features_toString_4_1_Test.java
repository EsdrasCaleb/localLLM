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
public class Features_toString_4_1_Test {

    @Test
    public void testToString() {
        Features features = new Features();
        features.setFeature(new String[] { "Feature1", "Feature2" });
        assertEquals("Number of Features = 2\nFeature - Feature1\nFeature - Feature2", features.toString());
        Features features2 = new Features();
        features2.setFeature(null);
        assertEquals("Number of Features = 0\n", features2.toString());
        Features features3 = new Features();
        features3.setFeature(new String[] {});
        assertEquals("Number of Features = 0\n", features3.toString());
    }
}
