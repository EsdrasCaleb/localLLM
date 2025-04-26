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

public class Features_toString_4_2_Test {

    @Test
    public void testToString() {
        Features features = new Features();
        ArrayList<String> featuresArrayList = new ArrayList<>();
        featuresArrayList.add("Feature 1");
        featuresArrayList.add("Feature 2");
        features.setFeature(featuresArrayList.toArray(new String[0]));
        String expectedString = "# of Feature = 2\nFeature - Feature 1\nFeature - Feature 2\n";
        assertEquals(expectedString, features.toString());
    }

    @Test
    public void testToStringEmpty() {
        Features features = new Features();
        features.setFeature(new String[0]);
        String expectedString = "Feature is null or size 0\n";
        assertEquals(expectedString, features.toString());
    }

    @Test
    public void testToStringNull() {
        Features features = new Features();
        features.setFeature(null);
        String expectedString = "Feature is null or size 0\n";
        assertEquals(expectedString, features.toString());
    }
}
