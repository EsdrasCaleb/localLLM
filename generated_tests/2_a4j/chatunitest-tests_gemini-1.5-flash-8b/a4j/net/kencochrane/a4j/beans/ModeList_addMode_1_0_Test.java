package net.kencochrane.a4j.beans;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class // Add more test cases as needed (e.g., for edge cases, different types of Mode)
// ...
ModeList_addMode_1_0_Test {

    @Test
    void addMode_withValidMode_addsModeToList() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        ModeList modeList = new ModeList();
        // Mock the Mode object
        Mode mode1 = Mockito.mock(Mode.class);
        Method addModeMethod = ModeList.class.getDeclaredMethod("addMode", Mode.class);
        // Important: make private method accessible
        addModeMethod.setAccessible(true);
        addModeMethod.invoke(modeList, mode1);
        ArrayList actualModes = modeList.getAllModes();
        assertEquals(1, actualModes.size());
        assertTrue(actualModes.contains(mode1));
    }

    @Test
    void addMode_withNullMode_doesNotThrowException() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        ModeList modeList = new ModeList();
        Mode nullMode = null;
        Method addModeMethod = ModeList.class.getDeclaredMethod("addMode", Mode.class);
        addModeMethod.setAccessible(true);
        addModeMethod.invoke(modeList, nullMode);
        ArrayList actualModes = modeList.getAllModes();
        // Important check for null input case
        assertEquals(0, actualModes.size());
    }
}

// Dummy Mode class (replace with your actual Mode class)
class Mode {
    // Add necessary fields and methods to your Mode class
    // For this test we only need a mock
}
