package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Features_toString_4_0_Test {

    @Test
    public void testToString() {
        Features features = new Features();
        features.setFeature(new String[] { "feature1", "feature2", "feature3" });
        String expectedOutput = "# of Feature = 3\n" + "Feature - feature1\n" + "Feature - feature2\n" + "Feature - feature3\n";
        assertEquals(expectedOutput, features.toString());
    }

    @Test
    public void testToStringWithNullFeatures() {
        Features features = new Features();
        String expectedOutput = "Feature is null or size 0\n";
        assertEquals(expectedOutput, features.toString());
    }

    @Test
    public void testToStringWithEmptyFeatures() {
        Features features = new Features();
        features.setFeature(new String[] {});
        String expectedOutput = "# of Feature = 0\n";
        assertEquals(expectedOutput, features.toString());
    }
}
