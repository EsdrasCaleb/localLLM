package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Platforms_toString_4_2_Test {

    @Test
    public void testToString() throws Exception {
        Platforms platforms = new Platforms();
        ArrayList<String> platformList = new ArrayList<String>();
        platformList.add("Platform 1");
        platformList.add("Platform 2");
        platformList.add("Platform 3");
        platforms.setPlatform(platformList.toArray(new String[0]));
        String expectedOutput = "# of Platforms = 3\n" + "Platform - Platform 1\n" + "Platform - Platform 2\n" + "Platform - Platform 3\n";
        Field field = Platforms.class.getDeclaredField("platform");
        field.setAccessible(true);
        field.set(platforms, platformList);
        String result = platforms.toString();
        Assertions.assertEquals(expectedOutput, result);
    }
}
