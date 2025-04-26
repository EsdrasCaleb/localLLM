// Features_toString_4_0_Test.java
package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class Features_toString_4_0_Test {

    @InjectMocks
    private Features features;

    @BeforeEach
    public void setup() {
        features = new Features();
    }

    @Test
    public void testToString_EmptyList() {
        String expected = "Feature is null or size 0";
        String actual = features.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToString_NonEmptyList() {
        features.setFeature(new String[] { "feature1", "feature2", "feature3" });
        String expected = "# of Feature = 3\nFeature - feature1\nFeature - feature2\nFeature - feature3";
        String actual = features.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToString_NullFeatures() {
        features.features = null;
        String expected = "Feature is null or size 0";
        String actual = features.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToString_NullFeaturesAndEmptyList() {
        features.features = null;
        String expected = "Feature is null or size 0";
        String actual = features.toString();
        assertEquals(expected, actual);
    }
}
