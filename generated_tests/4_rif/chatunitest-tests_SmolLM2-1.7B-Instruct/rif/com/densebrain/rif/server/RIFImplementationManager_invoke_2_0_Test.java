package com.densebrain.rif.server;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Method;
import java.rmi.RemoteException;
import java.util.Hashtable;
import java.util.Map;

public class RIFImplementationManager_invoke_2_0_Test {

    @Test
    public void testInvoke_NullInterfaceName() throws RemoteException {
        // Arrange
        String methodName = "methodName";
        Object[] params = new Object[1];
        params[0] = "value";
        // Act and Assert
        assertThrows(NullPointerException.class, () -> RIFImplementationManager.getInstance().invoke(null, methodName, params));
    }
}
