package com.densebrain.rif.server.transport;

import org.apache.axis2.AxisFault;
import org.apache.axis2.addressing.EndpointReference;
import org.apache.axis2.context.ConfigurationContext;
import org.apache.axis2.context.ConfigurationContextFactory;
import org.apache.axis2.description.AxisService;
import org.apache.axis2.rpc.receivers.RPCMessageReceiver;
import org.apache.axis2.transport.http.turnup.SimpleHTTPServer;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.mockito.junit.jupiter.MockitoExtension;
import java.rmi.RemoteException;
import java.util.List;
import java.util.LinkedList;
import static org.mockito.ArgumentMatchers.anyString;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import java.net.InetAddress;

class WebServiceContainer_newInstance_0_3_Test {

    @Test
    void newInstance_invalidPort() {
        assertThrows(IllegalArgumentException.class, () -> WebServiceContainer.newInstance("localhost", -1, "/mycontext"));
    }

    @Test
    void newInstance_emptyContextPath() {
        assertThrows(IllegalArgumentException.class, () -> WebServiceContainer.newInstance("localhost", 8080, ""));
    }
}
